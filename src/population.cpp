#include "population.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/shape.h"
#include "math/tensor.h"
#include "neural_network.h"

void Population::initializeWeights()
{
    for (auto &adapter : layer_adapters)
    {
        adapter.host_params.resize(adapter.param_shapes.size());
        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            size_t total = population_size * adapter.param_numel[p];
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

void Population::syncHostToDevice()
{
    for (auto &adapter : layer_adapters)
    {
        for (size_t p = 0; p < adapter.batched_params.size(); ++p)
        {
            if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU)
            {
                adapter.batched_params[p].setDataType(Data_Type::FLOAT16);
            }
            adapter.batched_params[p].uploadData(adapter.host_params[p]);
        }
    }
}

void Population::syncDeviceToHost() const
{
    for (const auto &adapter : layer_adapters)
    {
        for (size_t p = 0; p < adapter.batched_params.size(); ++p)
        {
            adapter.host_params[p] = adapter.batched_params[p].getData();
        }
    }
}

Population::Population(size_t _population_size,
                       const Neural_Network &_template_net,
                       size_t _state_dimension,
                       size_t _action_dimension,
                       Execution_Target _execution_target,
                       uint32_t _seed)
    : population_size(_population_size),
      state_dimension(_state_dimension),
      action_space_size(_action_dimension),
      execution_target(_execution_target),
      random_engine(_seed),
      is_mixed_precision_enabled(_template_net.isMixedPrecisionEnabled())
{
    if (_population_size == 0 || _state_dimension == 0 || _action_dimension == 0)
    {
        throw std::invalid_argument("Population dimensions must be greater than zero");
    }

    for (const auto &layer_ptr : _template_net.getLayers())
    {
        if (!layer_ptr)
        {
            throw std::invalid_argument("Population: Null layer in template network");
        }
        if (!layer_ptr->supportsPopulationBatch())
        {
            throw std::invalid_argument(std::format("Layer {} not supported in population mode",
                                                    magic_enum::enum_name(layer_ptr->getLayerType())));
        }

        Population_Layer_Adapter adapter;
        auto cloned_template = layer_ptr->clone();
        if (!cloned_template)
        {
            throw std::runtime_error("Population: Failed to clone template layer");
        }
        cloned_template->setExecutionTarget(_execution_target);
        cloned_template->setTrainingMode(false);
        cloned_template->setMixedPrecision(is_mixed_precision_enabled);
        adapter.template_layer = std::shared_ptr<const ILayer>(std::move(cloned_template));
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
    if (is_mixed_precision_enabled && _execution_target == Execution_Target::VULKAN_GPU)
    {
        batched_input_tensor.setDataType(Data_Type::FLOAT16);
    }
    batched_input_host.assign(_population_size * _state_dimension, 0.0f);

    initializeWeights();
}

size_t Population::selectAction(size_t individual_index, const std::vector<float> &state_data) const
{
    return selectAction(individual_index, state_data.data());
}

size_t Population::selectAction(size_t individual_index, const float *state_data) const
{
    if (individual_index >= population_size)
    {
        throw std::out_of_range("Individual index out of range in selectAction");
    }
    if (!state_data)
    {
        throw std::invalid_argument("Population::selectAction: Null state_data pointer");
    }

    Neural_Network individual_net = is_mixed_precision_enabled
        ? getIndividual(individual_index)
        : buildCpuIndividual(individual_index);
    const Execution_Target infer_target = individual_net.getExecutionTarget();
    std::vector<float> input_vector(state_data, state_data + state_dimension);
    Tensor input_tensor(1, state_dimension, std::move(input_vector), infer_target);
    Tensor output_tensor = individual_net.forward(input_tensor);
    if (infer_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    const auto &output_data = output_tensor.getData();
    if (output_data.empty())
    {
        return 0;
    }

    auto max_it = std::max_element(output_data.begin(), output_data.end());
    return static_cast<size_t>(std::distance(output_data.begin(), max_it));
}

size_t Population::selectAction(size_t individual_index, const Tensor &state_tensor) const
{
    if (individual_index >= population_size)
    {
        throw std::out_of_range("Individual index out of range in selectAction");
    }

    const bool use_cpu_fast_path = !is_mixed_precision_enabled &&
                                   state_tensor.getExecutionTarget() == Execution_Target::CPU;
    Neural_Network individual_net = use_cpu_fast_path
        ? buildCpuIndividual(individual_index)
        : getIndividual(individual_index);
    Tensor output_tensor = individual_net.forward(state_tensor);
    if (individual_net.getExecutionTarget() == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    const auto &output_data = output_tensor.getData();
    if (output_data.empty())
    {
        return 0;
    }

    auto max_it = std::max_element(output_data.begin(), output_data.end());
    return static_cast<size_t>(std::distance(output_data.begin(), max_it));
}

void Population::selectBatchActions(const float *states_flat,
                                    const size_t *active_indices,
                                    size_t active_count,
                                    size_t *actions_out) const
{
    if (active_count == 0)
    {
        return;
    }

    if (!states_flat || !actions_out)
    {
        throw std::invalid_argument("Population::selectBatchActions: Null pointer passed for states_flat or actions_out");
    }

    if (!active_indices && active_count > population_size)
    {
        throw std::out_of_range("Population::selectBatchActions: active_count exceeds population_size");
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
        for (size_t i = 0; i < active_count; ++i)
        {
            size_t agent_idx = active_indices ? active_indices[i] : i;
            if (agent_idx >= population_size)
            {
                throw std::out_of_range("Population::selectBatchActions: active agent index out of range");
            }
            const float *src = states_flat + (i * state_dimension);
            float *dst = batched_input_host.data() + (agent_idx * state_dimension);
            std::copy(src, src + state_dimension, dst);
        }
    }

    if (is_mixed_precision_enabled && execution_target == Execution_Target::VULKAN_GPU)
    {
        batched_input_tensor.setDataType(Data_Type::FLOAT16);
    }
    batched_input_tensor.uploadData(batched_input_host);

    Tensor current_tensor = batched_input_tensor;
    for (const auto &adapter : layer_adapters)
    {
        current_tensor = adapter.template_layer->forward(current_tensor, adapter.batched_params);
    }

    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        Execution_Engine::getInstance().executeGraph();
    }

    std::vector<float> action_values = current_tensor.getData();
    for (size_t i = 0; i < active_count; ++i)
    {
        size_t agent_idx = active_indices ? active_indices[i] : i;
        if (agent_idx >= population_size)
        {
            throw std::out_of_range("Population::selectBatchActions: active agent index out of range");
        }
        size_t offset = agent_idx * action_space_size;
        if (offset + action_space_size > action_values.size())
        {
            throw std::out_of_range("Population::selectBatchActions: action_values output size mismatch");
        }
        size_t best_action = 0;
        float max_value = action_values[offset];
        for (size_t a = 1; a < action_space_size; ++a)
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

void Population::evolve(const float *fitness_scores,
                        float elitism_ratio,
                        float mutation_rate,
                        float mutation_strength,
                        float crossover_rate,
                        size_t tournament_size,
                        size_t explicit_elite_count)
{
    if (population_size == 0)
    {
        return;
    }
    if (!fitness_scores)
    {
        throw std::invalid_argument("Population::evolve: fitness_scores is null");
    }
    if (!std::isfinite(elitism_ratio) || !std::isfinite(mutation_rate) ||
        !std::isfinite(mutation_strength) || mutation_strength < 0.0f ||
        !std::isfinite(crossover_rate))
    {
        throw std::invalid_argument("Population::evolve: Hyperparameters must be finite and mutation_strength >= 0");
    }

    auto scoreHigher = [fitness_scores](size_t a, size_t b) -> bool
    {
        const float fa = fitness_scores[a];
        const float fb = fitness_scores[b];
        const bool na = std::isnan(fa);
        const bool nb = std::isnan(fb);
        if (na != nb)
        {
            return !na;
        }
        if (na && nb)
        {
            return false;
        }
        return fa > fb;
    };

    std::vector<size_t> indices(population_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), scoreHigher);

    const float clamped_elitism = std::clamp(elitism_ratio, 0.0f, 1.0f);
    size_t elite_count = (explicit_elite_count > 0)
        ? std::min(explicit_elite_count, population_size)
        : std::min(population_size, std::max<size_t>(1, static_cast<size_t>(population_size * clamped_elitism)));

    std::uniform_int_distribution<size_t> pop_dist(0, population_size - 1);
    std::normal_distribution<float> norm_dist(0.0f, mutation_strength > 0.0f ? mutation_strength : 1.0f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    std::vector<std::vector<std::vector<float>>> prev_params;
    prev_params.reserve(layer_adapters.size());
    for (const auto &adapter : layer_adapters)
    {
        prev_params.push_back(adapter.host_params);
    }

    auto selectParentTournament = [&]() -> size_t {
        if (tournament_size <= 1 || population_size <= 1)
        {
            return pop_dist(random_engine);
        }
        size_t best_cand = pop_dist(random_engine);
        for (size_t t = 1; t < tournament_size; ++t)
        {
            size_t cand = pop_dist(random_engine);
            if (scoreHigher(cand, best_cand))
            {
                best_cand = cand;
            }
        }
        return best_cand;
    };

    for (size_t i = 0; i < elite_count; ++i)
    {
        size_t elite_src = indices[i];
        for (size_t l = 0; l < layer_adapters.size(); ++l)
        {
            auto &adapter = layer_adapters[l];
            for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                size_t len = adapter.param_numel[p];
                std::copy_n(prev_params[l][p].data() + elite_src * len, len,
                            adapter.host_params[p].data() + i * len);
            }
        }
    }

    for (size_t i = elite_count; i < population_size; ++i)
    {
        size_t target_idx = i;
        size_t parent_a = selectParentTournament();
        size_t parent_b = selectParentTournament();

        for (size_t l = 0; l < layer_adapters.size(); ++l)
        {
            auto &adapter = layer_adapters[l];
            for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
            {
                size_t len = adapter.param_numel[p];
                auto &host = adapter.host_params[p];
                const auto &prev = prev_params[l][p];

                if (!adapter.param_evolvable[p])
                {
                    std::copy_n(prev.data() + parent_a * len, len, host.data() + target_idx * len);
                    continue;
                }

                size_t dst = target_idx * len;
                size_t pa = parent_a * len;
                size_t pb = parent_b * len;

                for (size_t k = 0; k < len; ++k)
                {
                    float val = (prob_dist(random_engine) < crossover_rate) ? prev[pb + k] : prev[pa + k];
                    if (mutation_strength > 0.0f && prob_dist(random_engine) < mutation_rate)
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

Neural_Network Population::buildCpuIndividual(size_t index) const
{
    if (index >= population_size)
    {
        throw std::out_of_range("Individual index out of range");
    }

    Neural_Network individual_net(Execution_Target::CPU);
    for (const auto &adapter : layer_adapters)
    {
        auto layer_clone = adapter.template_layer->clone();
        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            size_t len = adapter.param_numel[p];
            std::vector<float> slice(adapter.host_params[p].begin() + index * len,
                                     adapter.host_params[p].begin() + (index + 1) * len);
            layer_clone->setPopulationParameter(p, std::move(slice));
        }
        individual_net.addLayer(std::move(layer_clone));
    }
    individual_net.setTrainingMode(false);
    individual_net.enableMixedPrecision(is_mixed_precision_enabled);
    return individual_net;
}

Neural_Network Population::getIndividual(size_t index) const
{
    if (index >= population_size)
    {
        throw std::out_of_range("Individual index out of range in getIndividual");
    }

    Neural_Network individual_net = buildCpuIndividual(index);
    individual_net.setExecutionTarget(execution_target);
    individual_net.setTrainingMode(false);
    individual_net.enableMixedPrecision(is_mixed_precision_enabled);
    return individual_net;
}

void Population::setIndividual(size_t index, const Neural_Network &network)
{
    if (index >= population_size)
    {
        throw std::out_of_range("Individual index out of range in setIndividual");
    }

    const auto &net_layers = network.getLayers();
    if (net_layers.size() != layer_adapters.size())
    {
        throw std::invalid_argument("Layer count mismatch between Neural_Network and Population");
    }

    for (size_t l = 0; l < layer_adapters.size(); ++l)
    {
        auto &adapter = layer_adapters[l];
        const auto &net_layer = net_layers[l];

        if (net_layer->getLayerType() != adapter.template_layer->getLayerType())
        {
            throw std::invalid_argument(std::format("Layer type mismatch at layer {}: expected {}, got {}",
                                                    l,
                                                    magic_enum::enum_name(adapter.template_layer->getLayerType()),
                                                    magic_enum::enum_name(net_layer->getLayerType())));
        }

        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            std::vector<float> param_data = net_layer->getPopulationParameter(p);
            size_t len = adapter.param_numel[p];
            if (param_data.size() != len)
            {
                throw std::invalid_argument(std::format("Parameter size mismatch at layer {}, param {}: expected {}, got {}",
                                                        l, p, len, param_data.size()));
            }

            std::copy(param_data.begin(), param_data.end(), adapter.host_params[p].begin() + index * len);
        }
    }

    syncHostToDevice();
}

size_t Population::getBestIndividualIndex(const float *fitness_scores) const noexcept
{
    if (population_size == 0 || fitness_scores == nullptr)
    {
        return 0;
    }
    size_t best_idx = 0;
    float max_fitness = fitness_scores[0];
    bool best_is_nan = std::isnan(max_fitness);
    for (size_t i = 1; i < population_size; ++i)
    {
        float val = fitness_scores[i];
        if (std::isnan(val))
        {
            continue;
        }
        if (best_is_nan || val > max_fitness)
        {
            max_fitness = val;
            best_idx = i;
            best_is_nan = false;
        }
    }
    return best_idx;
}

Neural_Network Population::getBestIndividual(const float *fitness_scores) const
{
    return getIndividual(getBestIndividualIndex(fitness_scores));
}

bool Population::saveIndividual(size_t individual_index, const std::string &file_path) const
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

    uint32_t file_version = 1;
    out_file.write(reinterpret_cast<const char *>(&file_version), sizeof(file_version));

    uint64_t state_dim_val = static_cast<uint64_t>(state_dimension);
    uint64_t action_dim_val = static_cast<uint64_t>(action_space_size);
    uint64_t layer_count_val = static_cast<uint64_t>(layer_adapters.size());

    out_file.write(reinterpret_cast<const char *>(&state_dim_val), sizeof(state_dim_val));
    out_file.write(reinterpret_cast<const char *>(&action_dim_val), sizeof(action_dim_val));
    out_file.write(reinterpret_cast<const char *>(&layer_count_val), sizeof(layer_count_val));

    for (const auto &adapter : layer_adapters)
    {
        Layer_Type layer_type = adapter.template_layer->getLayerType();
        out_file.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));

        uint64_t param_count = static_cast<uint64_t>(adapter.param_shapes.size());
        out_file.write(reinterpret_cast<const char *>(&param_count), sizeof(param_count));

        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            uint64_t len = static_cast<uint64_t>(adapter.param_numel[p]);
            out_file.write(reinterpret_cast<const char *>(&len), sizeof(len));
            const float *ptr = adapter.host_params[p].data() + individual_index * len;
            out_file.write(reinterpret_cast<const char *>(ptr), len * sizeof(float));
        }
    }

    out_file.flush();
    return static_cast<bool>(out_file);
}

bool Population::saveBestIndividual(const std::string &file_path, const float *fitness_scores) const
{
    return saveIndividual(getBestIndividualIndex(fitness_scores), file_path);
}

bool Population::saveIndividualAsInference(size_t individual_index, const std::string &file_path) const
{
    if (individual_index >= population_size)
    {
        return false;
    }
    try
    {
        Neural_Network individual_net = buildCpuIndividual(individual_index);
        individual_net.saveInference(file_path);
        return true;
    }
    catch (const std::exception &e)
    {
        Logger::logMessage(Input_Format{"Population::saveIndividualAsInference failed: {}", e.what()},
                           Log_Level::LOG_ERROR,
                           true,
                           0,
                           Log_Feature::MODEL_SERIALIZATION);
        return false;
    }
}

bool Population::saveBestIndividualAsInference(const std::string &file_path, const float *fitness_scores) const
{
    return saveIndividualAsInference(getBestIndividualIndex(fitness_scores), file_path);
}

bool Population::loadIndividual(size_t individual_index, const std::string &file_path)
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

    char magic_header[4] = {};
    if (!in_file.read(magic_header, 4) ||
        magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'I' || magic_header[3] != 'K')
    {
        return false;
    }

    uint32_t file_version = 0;
    if (!in_file.read(reinterpret_cast<char *>(&file_version), sizeof(file_version)) || file_version != 1)
    {
        return false;
    }

    uint64_t state_dim_val = 0;
    uint64_t action_dim_val = 0;
    uint64_t layer_count_val = 0;

    if (!in_file.read(reinterpret_cast<char *>(&state_dim_val), sizeof(state_dim_val)) ||
        !in_file.read(reinterpret_cast<char *>(&action_dim_val), sizeof(action_dim_val)) ||
        !in_file.read(reinterpret_cast<char *>(&layer_count_val), sizeof(layer_count_val)))
    {
        return false;
    }

    if (state_dim_val != state_dimension || action_dim_val != action_space_size || layer_count_val != layer_adapters.size())
    {
        return false;
    }

    std::vector<std::vector<std::vector<float>>> staging(layer_adapters.size());
    for (size_t l = 0; l < layer_adapters.size(); ++l)
    {
        const auto &adapter = layer_adapters[l];
        Layer_Type layer_type{};
        if (!in_file.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type)) ||
            layer_type != adapter.template_layer->getLayerType())
        {
            return false;
        }

        uint64_t param_count = 0;
        if (!in_file.read(reinterpret_cast<char *>(&param_count), sizeof(param_count)) ||
            param_count != adapter.param_shapes.size())
        {
            return false;
        }

        staging[l].resize(adapter.param_shapes.size());
        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            uint64_t len = 0;
            if (!in_file.read(reinterpret_cast<char *>(&len), sizeof(len)) ||
                len != adapter.param_numel[p])
            {
                return false;
            }

            if (len > 0)
            {
                staging[l][p].resize(static_cast<size_t>(len));
                if (!in_file.read(reinterpret_cast<char *>(staging[l][p].data()),
                                  static_cast<std::streamsize>(len * sizeof(float))))
                {
                    return false;
                }
            }
        }
    }

    for (size_t l = 0; l < layer_adapters.size(); ++l)
    {
        auto &adapter = layer_adapters[l];
        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            size_t len = adapter.param_numel[p];
            if (len > 0)
            {
                std::copy_n(staging[l][p].data(), len, adapter.host_params[p].data() + individual_index * len);
            }
        }
    }

    syncHostToDevice();
    return true;
}

bool Population::loadBestIndividual(const std::string &file_path)
{
    return loadIndividual(0, file_path);
}

bool Population::saveCheckpoint(const std::string &file_path, uint64_t generation, const float *fitness_scores) const
{
    std::ofstream out_file(file_path, std::ios::binary);
    if (!out_file.is_open())
    {
        return false;
    }

    const char magic_header[4] = {'N', 'N', 'P', 'K'};
    out_file.write(magic_header, 4);

    uint32_t file_version = 1;
    out_file.write(reinterpret_cast<const char *>(&file_version), sizeof(file_version));

    uint64_t generation_val = generation;
    uint64_t pop_size_val = static_cast<uint64_t>(population_size);
    uint64_t state_dim_val = static_cast<uint64_t>(state_dimension);
    uint64_t action_dim_val = static_cast<uint64_t>(action_space_size);
    uint64_t layer_count_val = static_cast<uint64_t>(layer_adapters.size());

    out_file.write(reinterpret_cast<const char *>(&generation_val), sizeof(generation_val));
    out_file.write(reinterpret_cast<const char *>(&pop_size_val), sizeof(pop_size_val));
    out_file.write(reinterpret_cast<const char *>(&state_dim_val), sizeof(state_dim_val));
    out_file.write(reinterpret_cast<const char *>(&action_dim_val), sizeof(action_dim_val));
    out_file.write(reinterpret_cast<const char *>(&layer_count_val), sizeof(layer_count_val));

    for (const auto &adapter : layer_adapters)
    {
        Layer_Type layer_type = adapter.template_layer->getLayerType();
        out_file.write(reinterpret_cast<const char *>(&layer_type), sizeof(layer_type));

        uint64_t param_count = static_cast<uint64_t>(adapter.param_shapes.size());
        out_file.write(reinterpret_cast<const char *>(&param_count), sizeof(param_count));

        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            uint64_t len = static_cast<uint64_t>(adapter.param_numel[p]);
            out_file.write(reinterpret_cast<const char *>(&len), sizeof(len));
            out_file.write(reinterpret_cast<const char *>(adapter.host_params[p].data()),
                           static_cast<std::streamsize>(adapter.host_params[p].size() * sizeof(float)));
        }
    }

    std::uint8_t has_fitness = (fitness_scores != nullptr) ? 1 : 0;
    out_file.write(reinterpret_cast<const char *>(&has_fitness), sizeof(has_fitness));
    if (has_fitness != 0)
    {
        out_file.write(reinterpret_cast<const char *>(fitness_scores),
                       static_cast<std::streamsize>(population_size * sizeof(float)));
    }

    out_file.flush();
    return static_cast<bool>(out_file);
}

bool Population::loadCheckpoint(const std::string &file_path, uint64_t &generation, float *fitness_scores_out)
{
    std::ifstream in_file(file_path, std::ios::binary);
    if (!in_file.is_open())
    {
        return false;
    }

    char magic_header[4] = {};
    if (!in_file.read(magic_header, 4) ||
        magic_header[0] != 'N' || magic_header[1] != 'N' || magic_header[2] != 'P' || magic_header[3] != 'K')
    {
        return false;
    }

    uint32_t file_version = 0;
    if (!in_file.read(reinterpret_cast<char *>(&file_version), sizeof(file_version)) || file_version != 1)
    {
        return false;
    }

    uint64_t generation_val = 0;
    uint64_t pop_size_val = 0;
    uint64_t state_dim_val = 0;
    uint64_t action_dim_val = 0;
    uint64_t layer_count_val = 0;

    if (!in_file.read(reinterpret_cast<char *>(&generation_val), sizeof(generation_val)) ||
        !in_file.read(reinterpret_cast<char *>(&pop_size_val), sizeof(pop_size_val)) ||
        !in_file.read(reinterpret_cast<char *>(&state_dim_val), sizeof(state_dim_val)) ||
        !in_file.read(reinterpret_cast<char *>(&action_dim_val), sizeof(action_dim_val)) ||
        !in_file.read(reinterpret_cast<char *>(&layer_count_val), sizeof(layer_count_val)))
    {
        return false;
    }

    if (pop_size_val != population_size || state_dim_val != state_dimension || action_dim_val != action_space_size || layer_count_val != layer_adapters.size())
    {
        return false;
    }

    std::vector<std::vector<std::vector<float>>> staging(layer_adapters.size());
    for (size_t l = 0; l < layer_adapters.size(); ++l)
    {
        const auto &adapter = layer_adapters[l];
        Layer_Type layer_type{};
        if (!in_file.read(reinterpret_cast<char *>(&layer_type), sizeof(layer_type)) ||
            layer_type != adapter.template_layer->getLayerType())
        {
            return false;
        }

        uint64_t param_count = 0;
        if (!in_file.read(reinterpret_cast<char *>(&param_count), sizeof(param_count)) ||
            param_count != adapter.param_shapes.size())
        {
            return false;
        }

        staging[l].resize(adapter.param_shapes.size());
        for (size_t p = 0; p < adapter.param_shapes.size(); ++p)
        {
            uint64_t len = 0;
            if (!in_file.read(reinterpret_cast<char *>(&len), sizeof(len)) ||
                len != adapter.param_numel[p])
            {
                return false;
            }

            size_t expected_count = adapter.host_params[p].size();
            staging[l][p].resize(expected_count);
            if (expected_count > 0 &&
                !in_file.read(reinterpret_cast<char *>(staging[l][p].data()),
                              static_cast<std::streamsize>(expected_count * sizeof(float))))
            {
                return false;
            }
        }
    }

    std::uint8_t has_fitness = 0;
    if (!in_file.read(reinterpret_cast<char *>(&has_fitness), sizeof(has_fitness)))
    {
        return false;
    }

    std::vector<float> staging_fitness;
    if (has_fitness != 0)
    {
        staging_fitness.resize(population_size, 0.0f);
        if (population_size > 0 &&
            !in_file.read(reinterpret_cast<char *>(staging_fitness.data()),
                          static_cast<std::streamsize>(population_size * sizeof(float))))
        {
            return false;
        }
    }

    generation = generation_val;
    for (size_t l = 0; l < layer_adapters.size(); ++l)
    {
        layer_adapters[l].host_params = std::move(staging[l]);
    }
    if (has_fitness != 0 && fitness_scores_out != nullptr)
    {
        std::copy(staging_fitness.begin(), staging_fitness.end(), fitness_scores_out);
    }

    syncHostToDevice();
    return true;
}

void Population::setExecutionTarget(Execution_Target _target)
{
    if (execution_target == _target)
    {
        return;
    }
    execution_target = _target;
    batched_input_tensor.setExecutionTarget(_target);
    if (is_mixed_precision_enabled)
    {
        batched_input_tensor.setDataType(_target == Execution_Target::VULKAN_GPU ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
    }
    for (auto &adapter : layer_adapters)
    {
        if (adapter.template_layer)
        {
            auto cloned = adapter.template_layer->clone();
            if (!cloned)
            {
                throw std::runtime_error("Population::setExecutionTarget: Failed to clone template layer");
            }
            cloned->setExecutionTarget(_target);
            adapter.template_layer = std::shared_ptr<const ILayer>(std::move(cloned));
        }
        for (auto &param : adapter.batched_params)
        {
            param.setExecutionTarget(_target);
            if (!is_mixed_precision_enabled || _target != Execution_Target::VULKAN_GPU)
            {
                param.setDataType(Data_Type::FLOAT32);
            }
        }
    }
    syncHostToDevice();
}

void Population::enableMixedPrecision(bool enable)
{
    is_mixed_precision_enabled = enable;
    if (execution_target == Execution_Target::VULKAN_GPU)
    {
        batched_input_tensor.setDataType(enable ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
        for (auto &adapter : layer_adapters)
        {
            for (auto &param : adapter.batched_params)
            {
                param.setDataType(enable ? Data_Type::FLOAT16 : Data_Type::FLOAT32);
            }
        }
        syncHostToDevice();
    }
    Logger::logMessage(Input_Format{"Population::enableMixedPrecision: Mixed precision {} for population size = {}, target = {}",
                                    enable ? "ENABLED (FLOAT16)" : "DISABLED (FLOAT32)", population_size,
                                    execution_target == Execution_Target::VULKAN_GPU ? "VULKAN_GPU" : "CPU"},
                       Log_Level::LOG_INFO,
                       true,
                       0,
                       Log_Feature::FP16_METRICS);
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "layer/ilayer.h"
#include "math/shape.h"
#include "math/tensor.h"
#include "neural_network.h"

inline size_t calculateNumel(const Shape &shape)
{
    size_t total = 1;
    for (size_t dimension : shape.getDimensions())
    {
        total *= dimension;
    }
    return total;
}

inline Shape prependPopulationDim(size_t population_size, const Shape &shape)
{
    std::vector<size_t> dimensions{population_size};
    dimensions.insert(dimensions.end(), shape.getDimensions().begin(), shape.getDimensions().end());
    return Shape(dimensions);
}

struct Population_Layer_Adapter
{
    const ILayer *template_layer = nullptr;
    std::vector<Shape> param_shapes;
    std::vector<bool> param_evolvable;
    std::vector<size_t> param_numel;
    std::vector<Tensor> batched_params;
    mutable std::vector<std::vector<float>> host_params;
};

class Population
{
private:
    size_t population_size = 0;
    size_t state_dimension = 0;
    size_t action_space_size = 0;
    Execution_Target execution_target = Execution_Target::CPU;
    mutable std::mt19937 random_engine;
    bool is_mixed_precision_enabled = false;

    std::vector<Population_Layer_Adapter> layer_adapters;

    mutable Tensor batched_input_tensor;
    mutable std::vector<float> batched_input_host;

    void initializeWeights();

public:
    void syncHostToDevice();
    void syncDeviceToHost() const;

    Population(size_t _population_size,
               const Neural_Network &_template_net,
               size_t _state_dimension,
               size_t _action_dimension,
               Execution_Target _execution_target = Execution_Target::CPU,
               uint32_t _seed = std::random_device{}());

    size_t selectAction(size_t individual_index, const std::vector<float> &state_data) const;
    size_t selectAction(size_t individual_index, const float *state_data) const;
    size_t selectAction(size_t individual_index, const Tensor &state_tensor) const;

    void selectBatchActions(const float *states_flat,
                            const size_t *active_indices,
                            size_t active_count,
                            size_t *actions_out) const;

    void evolve(const float *fitness_scores,
                float elitism_ratio = 0.15f,
                float mutation_rate = 0.15f,
                float mutation_strength = 0.15f,
                float crossover_rate = 0.5f,
                size_t tournament_size = 3,
                size_t explicit_elite_count = 0);

    Neural_Network getIndividual(size_t index) const;
    void setIndividual(size_t index, const Neural_Network &network);

    size_t getBestIndividualIndex(const float *fitness_scores) const noexcept;
    Neural_Network getBestIndividual(const float *fitness_scores) const;

    bool saveIndividual(size_t individual_index, const std::string &file_path) const;
    bool saveBestIndividual(const std::string &file_path, const float *fitness_scores) const;
    bool saveIndividualAsInference(size_t individual_index, const std::string &file_path) const;
    bool saveBestIndividualAsInference(const std::string &file_path, const float *fitness_scores) const;

    bool loadIndividual(size_t individual_index, const std::string &file_path);
    bool loadBestIndividual(const std::string &file_path);

    bool saveCheckpoint(const std::string &file_path, uint64_t generation = 0, const float *fitness_scores = nullptr) const;
    bool loadCheckpoint(const std::string &file_path, uint64_t &generation, float *fitness_scores_out = nullptr);

    const Tensor &getBatchedInputTensor() const noexcept { return batched_input_tensor; }
    Tensor &getBatchedInputTensor() noexcept { return batched_input_tensor; }
    const std::vector<Population_Layer_Adapter> &getLayerAdapters() const noexcept { return layer_adapters; }
    std::vector<Population_Layer_Adapter> &getLayerAdapters() noexcept { return layer_adapters; }
    const std::vector<float> &getBatchedInputHost() const noexcept { return batched_input_host; }
    std::vector<float> &getBatchedInputHost() noexcept { return batched_input_host; }
    size_t getActionSpaceSize() const noexcept { return action_space_size; }
    size_t getPopulationSize() const noexcept { return population_size; }
    size_t getStateDimension() const noexcept { return state_dimension; }
    Execution_Target getExecutionTarget() const noexcept { return execution_target; }

    void setBatchedInputTensor(Tensor _tensor) noexcept { batched_input_tensor = std::move(_tensor); }
    void setLayerAdapters(std::vector<Population_Layer_Adapter> _adapters) noexcept { layer_adapters = std::move(_adapters); }
    void setBatchedInputHost(std::vector<float> _host) noexcept { batched_input_host = std::move(_host); }
    void setActionSpaceSize(size_t _size) noexcept { action_space_size = _size; }
    void setPopulationSize(size_t _size) noexcept { population_size = _size; }
    void setStateDimension(size_t _dimension) noexcept { state_dimension = _dimension; }
    void setExecutionTarget(Execution_Target _target) noexcept { execution_target = _target; }
    void setMixedPrecision(bool enable) noexcept;
    void enableMixedPrecision(bool enable = true) noexcept { setMixedPrecision(enable); }
    bool isMixedPrecisionEnabled() const noexcept { return is_mixed_precision_enabled; }
};
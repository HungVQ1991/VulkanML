#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <fstream>

#include "layer/ilayer.h"
#include "math/tensor.h"

class Embedding_Layer : public ILayer
{
private:
    size_t vocab_size = 0;
    size_t embedding_dim = 0;
    Execution_Target execution_target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT32;

    Tensor weight;
    Tensor weight_gradient;
    Tensor last_indices_tensor;
    Tensor output_tensor;
    bool accumulate_on_backward_with_tied = false;
    void initializeWeights();

public:
    Embedding_Layer(size_t _vocab_size,
                    size_t _embedding_dim,
                    Execution_Target _execution_target = Execution_Target::CPU,
                    Data_Type _data_type = Data_Type::FLOAT32);
    Tensor forward(const std::vector<int32_t> &token_ids);
    Tensor forward(const std::vector<std::vector<int32_t>> &batch_token_ids);
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;

    bool hasParameters() const override { return true; }
    void resetGradient() override;
    void resetGradients() override;
    const Tensor &getWeights() const override { return weight; }
    const Tensor &getWeightsGradient() const override { return weight_gradient; }
    const Tensor &getInput() const override { return last_indices_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    Layer_Type getLayerType() const override { return Layer_Type::EMBEDDING; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    void setExecutionTarget(Execution_Target _execution_target) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;
    std::unique_ptr<ILayer> clone() const override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    size_t getParameterCount() const noexcept;

    size_t getVocabSize() const noexcept { return vocab_size; }
    size_t getEmbeddingDim() const noexcept { return embedding_dim; }
    const Tensor &getWeight() const noexcept;
    Tensor &getWeight() noexcept;
    const Tensor &getWeightGradient() const noexcept { return weight_gradient; }
    Tensor &getWeightGradient() noexcept { return weight_gradient; }
    void setAccumulateForTiedWeights(bool _enable) noexcept { accumulate_on_backward_with_tied = _enable; }
    bool isAccumulatingForTiedWeights() const noexcept { return accumulate_on_backward_with_tied; }

};

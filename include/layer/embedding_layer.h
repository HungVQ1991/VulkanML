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
#include "engine/execution_engine.h"

class Embedding_Layer : public ILayer
{
private:
    size_t vocab_size = 0;
    size_t embedding_dim = 0;
    Execution_Target execution_target = Execution_Target::CPU;
    Data_Type data_type = Data_Type::FLOAT32;

    Tensor weight;
    Tensor weight_gradient;
    mutable Tensor last_indices_tensor;
    mutable Tensor output_tensor;    void initializeWeights();


public:    Embedding_Layer(size_t _vocab_size,
                    size_t _embedding_dim,
                    Execution_Target _execution_target = Execution_Target::CPU,
                    Data_Type _data_type = Data_Type::FLOAT32);
    Tensor forward(const std::vector<int32_t> &token_ids) const;
    Tensor forward(const std::vector<std::vector<int32_t>> &batch_token_ids) const;
    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;


    bool hasParameters() const override { return true; }    void resetGradient() override;
    void resetGradients() override;
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;


    Layer_Type getLayerType() const override { return Layer_Type::EMBEDDING; }
    Execution_Target getExecutionTarget() const override { return execution_target; }    void setExecutionTarget(Execution_Target _execution_target) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;
    std::unique_ptr<ILayer> clone() const override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t) const override;
    size_t getParameterCount() const noexcept;

    size_t getVocabSize() const noexcept { return vocab_size; }
    size_t getEmbeddingDim() const noexcept { return embedding_dim; }    const Tensor &getWeight() const noexcept;
    Tensor &getWeight() noexcept;

};;

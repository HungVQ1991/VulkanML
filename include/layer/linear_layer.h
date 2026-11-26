#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

namespace gpu
{
class vector;
}

class Linear_Layer : public ILayer
{
public:
    static constexpr uint32_t TERNARY_INFERENCE_MAGIC = 0x5445524E;
    static constexpr uint32_t TERNARY_INFERENCE_MAGIC_V2 = 0x54455232;
    static constexpr uint32_t TERNARY_CHECKPOINT_MAGIC = 0x54455243;

private:
    Tensor weights;
    Tensor biases;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    Tensor weights_gradient_tensor;
    Tensor biases_gradient_tensor;

    Tensor input_tensor_fp16;
    Tensor output_gradient_fp16;
    Tensor input_gradient_tensor_fp16;

    Data_Type data_type = Data_Type::FLOAT32;
    Tensor master_weights;
    Tensor effective_weights;
    Tensor packed_weights;
    Tensor gamma_tensor;
    bool has_packed_inference = false;
    bool has_master_weights = true;
    bool use_bias = true;

    Tensor *tied_weight = nullptr;
    Tensor *tied_weight_grad = nullptr;
    bool tied_weight_transposed = false;
    Tensor tied_weight_t_scratch;
    Tensor tied_grad_fp32_scratch;
    Tensor unused_bias_grad_scratch;
    Tensor indexed_hidden_scratch;
    Tensor indexed_grad_scratch;

    std::shared_ptr<gpu::vector> gpu_partial_sums;
    std::shared_ptr<gpu::vector> gpu_gamma_buffer;

    size_t input_dimension = 0;
    size_t output_dimension = 0;
    float initialization_gain = 2.0f;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;
    bool cache_is_nd = false;
    Shape cache_orig_shape;

public:
    using ILayer::forward;

    Linear_Layer();
    Linear_Layer(size_t _input_dimension,
                 size_t _output_dimension,
                 Execution_Target _execution_target = Execution_Target::CPU,
                 float _initialization_gain = 2.0f,
                 Data_Type _data_type = Data_Type::FLOAT32,
                 bool _use_bias = true);

    ~Linear_Layer() noexcept override;

    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;

    std::unique_ptr<ILayer> clone() const override;
    void invalidateWeightCache() noexcept override;
    void setMixedPrecision(bool _enable) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void resetGradient() override;
    void resetGradients() override;

    void prepareInference() override;
    void quantizeToTernary() override;
    void dequantizeFromTernary() override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::function<float(std::mt19937 &)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;
    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;

    std::vector<Shape> getPopulationParameterDims() const override
    {
        return { { input_dimension, output_dimension }, { 1, output_dimension } };
    }

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<const Tensor *> getNoDecayParameters() const override
    {
        return (use_bias && is_trainable) ? std::vector<const Tensor *>{ &biases } : std::vector<const Tensor *>{};
    }

    Tensor forwardIndexed(const Tensor &hidden, const Tensor &target_indices);
    Tensor backwardIndexed(const Tensor &d_logits, const Tensor &target_indices, const Shape &full_hidden_shape);

    void tieWeights(Tensor *shared_weight, Tensor *shared_grad = nullptr, bool transposed = true);
    void untieWeights() noexcept;
    bool isWeightTied() const noexcept { return tied_weight != nullptr; }

    bool hasBias() const noexcept { return use_bias; }
    void setUseBias(bool _use_bias) noexcept { use_bias = _use_bias; }

    const Tensor &getWeightsGradient() const override { return weights_gradient_tensor; }
    const Tensor &getBiasesGradient() const noexcept { return biases_gradient_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getWeights() const override;
    const Tensor &getBiases() const override { return biases; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }

    const Tensor &getMasterWeights() const noexcept { return master_weights; }
    const Tensor &getEffectiveWeights() const noexcept { return effective_weights; }
    const Tensor &getPackedWeights() const noexcept { return packed_weights; }
    const Tensor &getGammaTensor() const noexcept { return gamma_tensor; }

    bool isTernary() const noexcept override { return data_type == Data_Type::TERNARY_INT2; }
    bool hasPackedInference() const noexcept { return has_packed_inference; }
    bool hasMasterWeights() const noexcept { return has_master_weights; }
    Data_Type getDataType() const noexcept { return data_type; }

    size_t getOutputDimension() const noexcept { return output_dimension; }
    size_t getInputDimension() const noexcept { return input_dimension; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    float getInitializationGain() const noexcept { return initialization_gain; }

    Layer_Type getLayerType() const noexcept override { return isTernary() ? Layer_Type::TERNARY_LINEAR : Layer_Type::LINEAR; }
    bool supportsPopulationBatch() const noexcept override { return !isTernary(); }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }

    void setWeights(const Tensor &_new_weights);
    void setMasterWeights(const Tensor &_new_weights);
    void setBiases(const Tensor &_new_biases);
    void setWeightsGradient(const Tensor &_tensor) { weights_gradient_tensor = _tensor; }
    void setBiasesGradient(const Tensor &_tensor) { biases_gradient_tensor = _tensor; }

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setInitializationGain(float _initialization_gain) noexcept { initialization_gain = _initialization_gain; }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
    void configureDimensionsForLoad(size_t _input_dim, size_t _output_dim) noexcept
    {
        input_dimension = _input_dim;
        output_dimension = _output_dim;
    }
    void copyStateTo(Linear_Layer &dst) const;

private:
    void ensureTernaryGpuBuffers();
    void quantizeEffectiveWeightsGpu();
    void quantizeEffectiveWeightsCpu();
    void packWeightsGpu();
    void packWeightsCpu();
};

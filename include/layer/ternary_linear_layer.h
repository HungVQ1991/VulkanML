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

class Ternary_Linear_Layer : public ILayer
{
private:
    Tensor master_weights;          // [K, N] FP32
    Tensor biases;                  // [1, N] FP32
    Tensor effective_weights;        // [K, N] FP16 / FP32 (Fake-quantized effective weights)
    Tensor packed_weights;           // [N, K] Logic TERNARY_INT2 (Packed 2-bit weights)
    Tensor gamma_tensor;             // [2] FP32 (gamma, inv_gamma)

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    Tensor master_weights_gradient_tensor; // [K, N] FP32
    Tensor biases_gradient_tensor;          // [1, N] FP32

    size_t input_dimension = 0;   // K
    size_t output_dimension = 0;  // N
    float initialization_gain = 2.0f;
    bool is_forward_completed = false;
    bool has_packed_inference = false;
    bool has_master_weights = true;
    Execution_Target execution_target = Execution_Target::CPU;
    bool cache_is_3d = false;
    Shape cache_orig_shape;

    // GPU Helper Buffers
    mutable std::shared_ptr<gpu::vector> gpu_partial_sums;
    mutable std::shared_ptr<gpu::vector> gpu_gamma_buffer;

public:
    using ILayer::forward;

    Ternary_Linear_Layer();
    Ternary_Linear_Layer(size_t _input_dimension,
                         size_t _output_dimension,
                         Execution_Target _execution_target = Execution_Target::CPU,
                         float _initialization_gain = 2.0f);

    ~Ternary_Linear_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;

    std::unique_ptr<ILayer> clone() const override;
    void invalidateWeightCache() noexcept override;
    void setMixedPrecision(bool _enable) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void resetGradient() override;
    void resetGradients() override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    // Inference Baking
    void prepareInference();

    std::function<float(std::mt19937 &)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;
    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;

    std::vector<Shape> getPopulationParameterDims() const override
    {
        return { {input_dimension, output_dimension}, {1, output_dimension} };
    }

    // Optimizer contract: ONLY register master_weights (FP32) and biases (FP32)
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<const Tensor *> getNoDecayParameters() const override { return { &biases }; }

    const Tensor &getWeightsGradient() const override { return master_weights_gradient_tensor; }
    const Tensor &getBiasesGradient() const noexcept { return biases_gradient_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getWeights() const override { return master_weights; }
    const Tensor &getBiases() const override { return biases; }
    const Tensor &getPackedWeights() const noexcept { return packed_weights; }
    const Tensor &getEffectiveWeights() const noexcept { return effective_weights; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }

    size_t getOutputDimension() const noexcept { return output_dimension; }
    size_t getInputDimension() const noexcept { return input_dimension; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    float getInitializationGain() const noexcept { return initialization_gain; }

    Layer_Type getLayerType() const noexcept override { return Layer_Type::TERNARY_LINEAR; }
    bool supportsPopulationBatch() const noexcept override { return false; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }
    bool hasPackedInference() const noexcept { return has_packed_inference; }

    void setMasterWeights(const Tensor &_new_weights);
    void setBiases(const Tensor &_new_biases);
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setInitializationGain(float _gain) noexcept { initialization_gain = _gain; }

private:
    void quantizeEffectiveWeightsGpu();
    void quantizeEffectiveWeightsCpu();
    void packWeightsGpu();
    void packWeightsCpu();
};

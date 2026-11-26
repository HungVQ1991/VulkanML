#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "layer/dropout.h"
#include "layer/ilayer.h"
#include "layer/linear_layer.h"
#include "math/tensor.h"

class LoRA_Linear_Layer : public ILayer
{
private:
    Linear_Layer base_layer;
    Linear_Layer lora_a;
    Linear_Layer lora_b;
    std::unique_ptr<Dropout_Layer> lora_dropout;

    size_t rank = 0;
    float alpha = 1.0f;
    float dropout_rate = 0.0f;
    bool lora_enabled = false;
    bool is_merged = false;
    bool is_training_mode = false;
    bool cache_has_dropout = false;

    Tensor cache_scaled_delta;
    Tensor cache_casted_delta;
    Tensor cache_output;
    Tensor cache_dy_scaled;
    Tensor cache_dx_lora;
    Tensor cache_dx_total;

    void initLoRAParams(size_t new_rank, float new_alpha, float new_dropout);

public:
    using ILayer::forward;

    LoRA_Linear_Layer();
    LoRA_Linear_Layer(size_t _input_dimension,
                      size_t _output_dimension,
                      size_t _rank = 0,
                      float _alpha = 1.0f,
                      float _dropout_rate = 0.0f,
                      Execution_Target _execution_target = Execution_Target::CPU,
                      float _initialization_gain = 2.0f,
                      Data_Type _data_type = Data_Type::FLOAT32,
                      bool _use_bias = true);
    explicit LoRA_Linear_Layer(Linear_Layer _base_layer,
                               size_t _rank = 0,
                               float _alpha = 1.0f,
                               float _dropout_rate = 0.0f);

    LoRA_Linear_Layer(const LoRA_Linear_Layer &other);
    LoRA_Linear_Layer &operator=(const LoRA_Linear_Layer &other);
    LoRA_Linear_Layer(LoRA_Linear_Layer &&other) noexcept = default;
    LoRA_Linear_Layer &operator=(LoRA_Linear_Layer &&other) noexcept = default;
    ~LoRA_Linear_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;

    void enableLoRA(size_t _rank, float _alpha, float _dropout_rate = 0.0f, bool _freeze_base = true);
    void disableLoRA();
    bool isLoRAEnabled() const noexcept { return lora_enabled && rank > 0; }
    bool isMerged() const noexcept { return is_merged; }
    void mergeWeights();
    void unmergeWeights();

    void setTrainable(bool _trainable) noexcept override;
    void setBaseTrainable(bool _trainable) noexcept { base_layer.setTrainable(_trainable); }
    bool isBaseTrainable() const noexcept { return base_layer.isTrainable(); }

    Linear_Layer &getBaseLayer() noexcept { return base_layer; }
    const Linear_Layer &getBaseLayer() const noexcept { return base_layer; }
    Linear_Layer &getLoRAA() noexcept { return lora_a; }
    const Linear_Layer &getLoRAA() const noexcept { return lora_a; }
    Linear_Layer &getLoRAB() noexcept { return lora_b; }
    const Linear_Layer &getLoRAB() const noexcept { return lora_b; }

    size_t getRank() const noexcept { return rank; }
    float getAlpha() const noexcept { return alpha; }
    float getDropoutRate() const noexcept { return dropout_rate; }
    float getScaling() const noexcept { return (rank > 0) ? (alpha / static_cast<float>(rank)) : 0.0f; }

    size_t getInputDimension() const noexcept { return base_layer.getInputDimension(); }
    size_t getOutputDimension() const noexcept { return base_layer.getOutputDimension(); }
    bool hasBias() const noexcept { return base_layer.hasBias(); }
    void setUseBias(bool _use_bias) noexcept { base_layer.setUseBias(_use_bias); }

    bool hasParameters() const override { return base_layer.hasParameters(); }
    bool supportsStaticGraph() const noexcept override { return (!isLoRAEnabled() || is_merged) && base_layer.supportsStaticGraph(); }
    void resetGradient() override;
    void resetGradients() override;
    std::unique_ptr<ILayer> clone() const override;

    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;

    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;
    std::vector<const Tensor *> getNoDecayParameters() const override { return base_layer.getNoDecayParameters(); }

    const Tensor &getWeights() const override { return base_layer.getWeights(); }
    const Tensor &getBiases() const override { return base_layer.getBiases(); }
    const Tensor &getWeightsGradient() const override { return base_layer.getWeightsGradient(); }
    const Tensor &getInput() const override { return base_layer.getInput(); }
    const Tensor &getOutput() const override { return isLoRAEnabled() && !is_merged ? cache_output : base_layer.getOutput(); }

    Execution_Target getExecutionTarget() const override { return base_layer.getExecutionTarget(); }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::LORA_LINEAR; }

    void setExecutionTarget(Execution_Target _execution_target) override;
    void setAccumulated(bool _is_accumulated) noexcept override;
    void setTrainingMode(bool _is_training) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;

    void prepareInference() override { base_layer.prepareInference(); }
    void quantizeToTernary() override;
    void dequantizeFromTernary() override;
    bool isTernary() const noexcept override { return base_layer.isTernary(); }
};

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ilayer.h"
#include "math/tensor.h"

class Instance_Norm_2d_Layer : public ILayer
{
private:
    uint32_t input_height = 0;
    uint32_t input_width = 0;
    uint32_t channels = 0;
    float epsilon = 1e-5f;
    Execution_Target execution_target = Execution_Target::CPU;

    Tensor gamma;
    Tensor beta;
    Tensor gamma_gradient_tensor;
    Tensor beta_gradient_tensor;

    Tensor saved_mean;
    Tensor saved_inv_std;
    Tensor normalized_input;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;

public:
    using ILayer::forward;

    Instance_Norm_2d_Layer(
        uint32_t _height,
        uint32_t _width,
        uint32_t _channels,
        float _epsilon = 1e-5f,
        Execution_Target _execution_target = Execution_Target::CPU);

    ~Instance_Norm_2d_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor &_batched_input, const std::vector<Tensor> &_batched_params) const override;
    std::unique_ptr<ILayer> clone() const override;
    Tensor backward(const Tensor &_output_gradient) override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;

    std::vector<Shape> getPopulationParameterDims() const noexcept override { return { Shape{1, channels}, Shape{1, channels} }; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return { true, true }; }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    const Tensor &getWeightsGradient() const override { return gamma_gradient_tensor; }
    const Tensor &getGammaGradient() const noexcept { return gamma_gradient_tensor; }
    const Tensor &getBetaGradient() const noexcept { return beta_gradient_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getNormalizedInput() const noexcept { return normalized_input; }
    const Tensor &getSavedMean() const noexcept { return saved_mean; }
    const Tensor &getSavedInvStd() const noexcept { return saved_inv_std; }

    const Tensor &getWeights() const override { return gamma; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getBiases() const override { return beta; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getGamma() const noexcept { return gamma; }
    const Tensor &getBeta() const noexcept { return beta; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::INSTANCE_NORM_2D; }
    uint32_t getInputHeight() const noexcept { return input_height; }
    uint32_t getInputWidth() const noexcept { return input_width; }
    uint32_t getChannels() const noexcept { return channels; }

    float getEpsilon() const noexcept { return epsilon; }
    bool supportsPopulationBatch() const override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }

    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;
    void setGammaGradient(const Tensor &_tensor) { gamma_gradient_tensor = _tensor; }
    void setBetaGradient(const Tensor &_tensor) { beta_gradient_tensor = _tensor; }
    void setSavedMean(const Tensor &_tensor) { saved_mean = _tensor; }
    void setSavedInvStd(const Tensor &_tensor) { saved_inv_std = _tensor; }
    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setNormalizedInput(const Tensor &_tensor) { normalized_input = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }
    void setGamma(const Tensor &_tensor) { gamma = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setBeta(const Tensor &_tensor) { beta = _tensor; }
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setInputHeight(uint32_t _height) noexcept { input_height = _height; }
    void setInputWidth(uint32_t _width) noexcept { input_width = _width; }
    void setChannels(uint32_t _channels) noexcept { channels = _channels; }
    void setEpsilon(float _epsilon) noexcept { epsilon = _epsilon; }
    void setIsForwardCompleted(bool _is_completed) noexcept { is_forward_completed = _is_completed; }
    void setTrainingMode(bool /*_is_training*/) override {}
};

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class Batch_Norm_Layer : public ILayer
{
private:
    size_t input_dimension = 0;
    float epsilon = 1e-5f;
    float momentum = 0.1f;
    bool is_training = true;

    Tensor gamma;
    Tensor beta;
    Tensor gamma_gradient_tensor;
    Tensor beta_gradient_tensor;

    Tensor running_mean;
    Tensor running_variance;
    Tensor batch_mean;
    Tensor batch_variance;
    Tensor normalized_input;

    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;    void initializeParameters();


public:
    using ILayer::forward;    explicit Batch_Norm_Layer(size_t _dimension,
                              float _epsilon = 1e-5f,
                              float _momentum = 0.1f,
                              Execution_Target _execution_target = Execution_Target::CPU);


    ~Batch_Norm_Layer() noexcept override = default;    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;
    std::unique_ptr<ILayer> clone() const override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;

    std::vector<Shape> getPopulationParameterDims() const override { return { Shape{ 1, input_dimension }, Shape{ 1, input_dimension }, Shape{ 1, input_dimension }, Shape{ 1, input_dimension } }; }
    std::vector<bool> getPopulationParameterIsEvolvable() const override { return {true, true, false, false}; }    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    const Tensor &getWeightsGradient() const override { return gamma_gradient_tensor; }
    const Tensor &getGammaGradient() const noexcept { return gamma_gradient_tensor; }
    const Tensor &getBetaGradient() const noexcept { return beta_gradient_tensor; }
    const Tensor &getRunningVariance() const noexcept { return running_variance; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }    const Tensor &getNormalizedInput() const noexcept;
    const Tensor &getBatchVariance() const noexcept;

    const Tensor &getRunningMean() const noexcept { return running_mean; }    const Tensor &getBatchMean() const noexcept;

    const Tensor &getWeights() const override { return gamma; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getBiases() const override { return beta; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getGamma() const noexcept { return gamma; }
    const Tensor &getBeta() const noexcept { return beta; }
    size_t getInputDimension() const noexcept { return input_dimension; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::BATCH_NORM; }
    float getMomentum() const noexcept { return momentum; }
    float getEpsilon() const noexcept { return epsilon; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }    bool isTraining() const noexcept;
    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;
    void setGammaGradient(const Tensor &_tensor);
    void setBetaGradient(const Tensor &_tensor);
    void setRunningVariance(const Tensor &_tensor);

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }    void setNormalizedInput(const Tensor &_tensor);
    void setBatchVariance(const Tensor &_tensor);
    void setRunningMean(const Tensor &_tensor);
    void setBatchMean(const Tensor &_tensor);

    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }    void setGamma(const Tensor &_tensor);

    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }    void setBeta(const Tensor &_tensor);
    void setInputDimension(size_t _dimension) noexcept;
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setMomentum(float _momentum) noexcept;
    void setEpsilon(float _epsilon) noexcept;
    void setIsForwardCompleted(bool _is_completed) noexcept;
    void setTrainingMode(bool _is_training) override;
    void setIsTraining(bool _is_training) noexcept;

};;
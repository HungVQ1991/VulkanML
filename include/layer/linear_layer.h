#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "ilayer.h"
#include "math/tensor.h"

class Linear_Layer : public ILayer
{
private:
    Tensor weights;
    Tensor biases;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    Tensor weights_gradient_tensor;
    Tensor biases_gradient_tensor;

    size_t input_dimension = 0;
    size_t output_dimension = 0;
    float initialization_gain = 0.02f;
    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;
    bool cache_is_3d = false;
    Shape cache_orig_shape;

public:
    using ILayer::forward;    Linear_Layer();
    Linear_Layer(size_t _input_dimension,
                 size_t _output_dimension,
                 Execution_Target _execution_target = Execution_Target::CPU,
                 float _initialization_gain = 2.0f,
                 Data_Type _data_type = Data_Type::FLOAT32);


    ~Linear_Layer() noexcept override = default;    Tensor forward(const Tensor &_input_tensor) override;
    Tensor forward(const Tensor& _batched_input, const std::vector<Tensor>& _batched_params) const override;
    Tensor backward(const Tensor &_output_gradient) override;
    std::unique_ptr<ILayer> clone() const override;
    void invalidateWeightCache() noexcept override;
    void setMixedPrecision(bool _enable) noexcept override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;
    std::function<float(std::mt19937&)> getPopulationParameterInitializer(size_t param_index) const override;
    std::vector<float> getPopulationParameter(size_t param_index) const override;

    std::vector<Shape> getPopulationParameterDims() const override 
    {
        std::vector<Shape> result;
        result.push_back({input_dimension, output_dimension});
        result.push_back({1, output_dimension});
        return result;
    }    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    const Tensor &getWeightsGradient() const override { return weights_gradient_tensor; }
    const Tensor &getBiasesGradient() const noexcept { return biases_gradient_tensor; }
    const Tensor &getInputGradient() const noexcept { return input_gradient_tensor; }
    const Tensor &getWeights() const override { return weights; }
    const Tensor &getBiases() const override { return biases; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    size_t getOutputDimension() const noexcept { return output_dimension; }
    size_t getInputDimension() const noexcept { return input_dimension; }
    Execution_Target getExecutionTarget() const override { return execution_target; }    float getInitializationGain() const noexcept;

    Layer_Type getLayerType() const noexcept override { return Layer_Type::LINEAR; }
    bool supportsPopulationBatch() const noexcept override { return true; }
    bool isForwardCompleted() const noexcept { return is_forward_completed; }
    bool hasParameters() const noexcept override { return true; }    void setPopulationParameter(size_t param_index, std::vector<float> flat_data) override;
    void setWeights(const Tensor &_new_weights);
    void setBiases(const Tensor &_new_biases);
    void setWeightsGradient(const Tensor &_tensor);
    void setBiasesGradient(const Tensor &_tensor);

    void setInputGradient(const Tensor &_tensor) { input_gradient_tensor = _tensor; }
    void setInput(const Tensor &_tensor) { input_tensor = _tensor; }
    void setOutput(const Tensor &_tensor) { output_tensor = _tensor; }    void setOutputDimension(size_t _output_dimension) noexcept;
    void setInputDimension(size_t _input_dimension) noexcept;
    void setExecutionTarget(Execution_Target _new_execution_target) override;
    void setInitializationGain(float _initialization_gain) noexcept;
    void setIsForwardCompleted(bool _is_completed) noexcept;

};;
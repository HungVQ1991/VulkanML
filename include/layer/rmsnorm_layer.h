#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/execution_engine.h"
#include "helper/logger.h"
#include "helper/magic_enum.hpp"
#include "layer/ilayer.h"
#include "math/tensor.h"

class RMSNorm_Layer : public ILayer
{
private:
    size_t dimension = 0;
    float epsilon = 1e-5f;

    Tensor gamma;
    Tensor gamma_gradient_tensor;

    Tensor inv_rms;
    Tensor input_tensor;
    Tensor output_tensor;
    Tensor input_gradient_tensor;

    bool is_forward_completed = false;
    Execution_Target execution_target = Execution_Target::CPU;
    bool cache_is_3d = false;
    Shape cache_orig_shape;    void initializeParameters();


public:
    using ILayer::forward;    explicit RMSNorm_Layer(size_t _dimension,
                           float _epsilon = 1e-5f,
                           Execution_Target _execution_target = Execution_Target::CPU,
                           Data_Type _data_type = Data_Type::FLOAT32);


    ~RMSNorm_Layer() noexcept override = default;    Tensor forward(const Tensor &_input_tensor) override;
    Tensor backward(const Tensor &_output_gradient) override;
    std::unique_ptr<ILayer> clone() const override;
    void resetGradient() override;
    void resetGradients() override;
    void saveConfiguration(std::ofstream &_output_file_stream) const override;
    void saveInference(std::ofstream &_output_file_stream) const override;
    void loadInference(std::ifstream &_input_file_stream) override;
    void saveCheckpoint(std::ofstream &_output_file_stream) const override;
    void loadCheckpoint(std::ifstream &_input_file_stream) override;


    bool hasParameters() const noexcept override { return true; }    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override;

    const Tensor &getWeights() const override { return gamma; }
    const Tensor &getWeightsGradient() const override { return gamma_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getGamma() const noexcept { return gamma; }
    const Tensor &getGammaGradient() const noexcept { return gamma_gradient_tensor; }    size_t getDimension() const noexcept;

    float getEpsilon() const noexcept { return epsilon; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::RMSNORM; }    void setExecutionTarget(Execution_Target _new_target) override;
    void setMixedPrecision(bool _enable) noexcept override;
    void invalidateWeightCache() noexcept override;

};;

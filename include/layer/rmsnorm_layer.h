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
    Shape cache_orig_shape;

    void initializeParameters()
    {
        std::vector<float> gamma_data(dimension, 1.0f);
        gamma = Tensor(1, dimension, std::move(gamma_data), execution_target);
        gamma_gradient_tensor = Tensor(1, dimension, execution_target);
    }

public:
    using ILayer::forward;

    explicit RMSNorm_Layer(size_t _dimension,
                           float _epsilon = 1e-5f,
                           Execution_Target _execution_target = Execution_Target::CPU,
                           Data_Type _data_type = Data_Type::FLOAT32)
        : dimension(_dimension),
          epsilon(_epsilon),
          gamma(0, 0, _execution_target),
          gamma_gradient_tensor(0, 0, _execution_target),
          inv_rms(0, 0, _execution_target),
          input_tensor(0, 0, _execution_target),
          output_tensor(0, 0, _execution_target),
          input_gradient_tensor(0, 0, _execution_target),
          is_forward_completed(false),
          execution_target(_execution_target)
    {
        if (_data_type == Data_Type::FLOAT16)
        {
            is_mixed_precision_enabled = true;
        }
        initializeParameters();
    }

    ~RMSNorm_Layer() noexcept override = default;

    Tensor forward(const Tensor &_input_tensor) override
    {
        size_t in_features = (_input_tensor.getShape().getRank() >= 2)
                             ? _input_tensor.getShape()[_input_tensor.getShape().getRank() - 1]
                             : _input_tensor.getColumns();
        if (in_features != dimension)
        {
            Logger::logMessage(Input_Format{"RMSNorm_Layer::forward: Dimension mismatch (expected {}, got {})",
                                            dimension, in_features},
                               Log_Level::LOG_ERROR, true, 0, Log_Feature::ACTIVATION_COMPUTE);
            throw std::invalid_argument(std::format("RMSNorm input dimension mismatch (expected {}, got {})", dimension, in_features));
        }

        cache_orig_shape = _input_tensor.getShape();
        cache_is_3d = (cache_orig_shape.getRank() == 3);

        Tensor eff_input = _input_tensor;
        if (cache_is_3d)
        {
            eff_input.reshape(Shape{ cache_orig_shape[0] * cache_orig_shape[1], dimension });
        }

        input_tensor = eff_input;
        input_tensor.rmsNormForward(gamma, inv_rms, output_tensor, epsilon);

        if (cache_is_3d)
        {
            output_tensor.reshape(cache_orig_shape);
        }

        is_forward_completed = true;
        return output_tensor;
    }

    Tensor backward(const Tensor &_output_gradient) override
    {
        if (!is_forward_completed)
        {
            throw std::logic_error("RMSNorm_Layer::backward called before forward");
        }

        Tensor eff_grad = _output_gradient;
        if (cache_is_3d && eff_grad.getShape().getRank() == 3)
        {
            eff_grad.reshape(Shape{ cache_orig_shape[0] * cache_orig_shape[1], dimension });
        }

        input_tensor.rmsNormBackward(eff_grad, gamma, inv_rms,
                                     gamma_gradient_tensor, input_gradient_tensor, is_accumulated);

        if (cache_is_3d)
        {
            input_gradient_tensor.reshape(cache_orig_shape);
        }

        return input_gradient_tensor;
    }

    std::unique_ptr<ILayer> clone() const override
    {
        auto copy = std::make_unique<RMSNorm_Layer>(dimension, epsilon, execution_target);
        copy->setMixedPrecision(is_mixed_precision_enabled);
        return copy;
    }

    void resetGradient() override
    {
        is_forward_completed = false;
    }

    void resetGradients() override
    {
        resetGradient();
        gamma_gradient_tensor.zero();
    }

    void saveConfiguration(std::ofstream &_output_file_stream) const override
    {
        uint32_t feature_count = static_cast<uint32_t>(dimension);
        _output_file_stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
        _output_file_stream.write(reinterpret_cast<const char *>(&epsilon), sizeof(epsilon));
    }

    void saveInference(std::ofstream &_output_file_stream) const override
    {
        gamma.saveTensor(_output_file_stream);
    }

    void loadInference(std::ifstream &_input_file_stream) override
    {
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        gamma.invalidateFp16Cache();
    }

    void saveCheckpoint(std::ofstream &_output_file_stream) const override
    {
        gamma.saveTensor(_output_file_stream);
        gamma_gradient_tensor.saveTensor(_output_file_stream);
    }

    void loadCheckpoint(std::ifstream &_input_file_stream) override
    {
        gamma = Tensor::loadTensor(_input_file_stream, execution_target);
        gamma_gradient_tensor = Tensor::loadTensor(_input_file_stream, execution_target);
        gamma.invalidateFp16Cache();
    }

    bool hasParameters() const noexcept override { return true; }
    std::vector<std::pair<Tensor *, Tensor *>> getParametersAndGradients() override
    {
        return {{&gamma, &gamma_gradient_tensor}};
    }
    const Tensor &getWeights() const override { return gamma; }
    const Tensor &getWeightsGradient() const override { return gamma_gradient_tensor; }
    const Tensor &getInput() const override { return input_tensor; }
    const Tensor &getOutput() const override { return output_tensor; }
    const Tensor &getGamma() const noexcept { return gamma; }
    const Tensor &getGammaGradient() const noexcept { return gamma_gradient_tensor; }
    size_t getDimension() const noexcept { return dimension; }
    float getEpsilon() const noexcept { return epsilon; }
    Execution_Target getExecutionTarget() const override { return execution_target; }
    Layer_Type getLayerType() const noexcept override { return Layer_Type::RMSNORM; }

    void setExecutionTarget(Execution_Target _new_target) override
    {
        if (execution_target == _new_target) return;
        execution_target = _new_target;
        gamma.setExecutionTarget(_new_target);
        gamma_gradient_tensor.setExecutionTarget(_new_target);
        inv_rms.setExecutionTarget(_new_target);
        input_tensor.setExecutionTarget(_new_target);
        output_tensor.setExecutionTarget(_new_target);
        input_gradient_tensor.setExecutionTarget(_new_target);
    }

    void setMixedPrecision(bool _enable) noexcept override
    {
        is_mixed_precision_enabled = _enable;
        gamma.invalidateFp16Cache();
    }

    void invalidateWeightCache() noexcept override
    {
        gamma.invalidateFp16Cache();
    }
};
